'use strict';

module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.createTable('DummyVariable', {
      Id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      RunId: {
        type: Sequelize.INTEGER,
        references: {
          model: 'RunDetail',
          key: 'Id'
        },
        onUpdate: 'cascade',
        onDelete: 'cascade'
      },
      Name: Sequelize.STRING,
      Coefficient: Sequelize.FLOAT,
      Pval: Sequelize.FLOAT
    });
  },

  down: (queryInterface, Sequelize) => {
    return queryInterface.dropTable('DummyVariable');
  }
};
